import { BadRequestException, Injectable } from '@nestjs/common';
import { AssetMapOptions, AssetResponseDto, mapAsset } from 'src/dtos/asset-response.dto';
import { AuthDto } from 'src/dtos/auth.dto';
import { mapPerson, PersonResponseDto } from 'src/dtos/person.dto';
import {
  mapPlaces,
  MetadataSearchDto,
  PlacesResponseDto,
  RandomSearchDto,
  SearchPeopleDto,
  SearchPlacesDto,
  SearchResponseDto,
  SearchSuggestionRequestDto,
  SearchSuggestionType,
  SmartSearchDto,
} from 'src/dtos/search.dto';
import { AssetEntity } from 'src/entities/asset.entity';
import { AssetOrder } from 'src/enum';
import { SearchExploreItem } from 'src/repositories/search.repository';
import { BaseService } from 'src/services/base.service';
import { getMyPartnerIds } from 'src/utils/asset.util';
import { isSmartSearchEnabled } from 'src/utils/misc';


// @todo remove, for POC only
type SmartSearchQuery = {
  similarToAssetIds: string[];
  searchText: string;
};

@Injectable()
export class SearchService extends BaseService {
  async searchPerson(auth: AuthDto, dto: SearchPeopleDto): Promise<PersonResponseDto[]> {
    const people = await this.personRepository.getByName(auth.user.id, dto.name, { withHidden: dto.withHidden });
    return people.map((person) => mapPerson(person));
  }

  async searchPlaces(dto: SearchPlacesDto): Promise<PlacesResponseDto[]> {
    const places = await this.searchRepository.searchPlaces(dto.name);
    return places.map((place) => mapPlaces(place));
  }

  async getExploreData(auth: AuthDto): Promise<SearchExploreItem<AssetResponseDto>[]> {
    const options = { maxFields: 12, minAssetsPerField: 5 };
    const cities = await this.assetRepository.getAssetIdByCity(auth.user.id, options);
    const assets = await this.assetRepository.getByIdsWithAllRelations(cities.items.map(({ data }) => data));
    const items = assets.map((asset) => ({ value: asset.exifInfo!.city!, data: mapAsset(asset, { auth }) }));
    return [{ fieldName: cities.fieldName, items }];
  }

  async searchMetadata(auth: AuthDto, dto: MetadataSearchDto): Promise<SearchResponseDto> {
    let checksum: Buffer | undefined;
    const userIds = await this.getUserIdsToSearch(auth);

    if (dto.checksum) {
      const encoding = dto.checksum.length === 28 ? 'base64' : 'hex';
      checksum = Buffer.from(dto.checksum, encoding);
    }

    const page = dto.page ?? 1;
    const size = dto.size || 250;
    const { hasNextPage, items } = await this.searchRepository.searchMetadata(
      { page, size },
      {
        ...dto,
        checksum,
        userIds,
        orderDirection: dto.order ?? AssetOrder.DESC,
      },
    );

    return this.mapResponse(items, hasNextPage ? (page + 1).toString() : null, { auth });
  }

  async searchRandom(auth: AuthDto, dto: RandomSearchDto): Promise<AssetResponseDto[]> {
    const userIds = await this.getUserIdsToSearch(auth);
    const items = await this.searchRepository.searchRandom(dto.size || 250, { ...dto, userIds });
    return items.map((item) => mapAsset(item, { auth }));
  }

  /**
   * Parses a user input into individual search components
   * @param smartSearchQuery The Query
   *
   * Example
   * smartSearchQuery: green st patricks day, green clothing similarTo:<uuid1> similarTo:<uuid2>
   * will return
   * {
   *  assetIds: [<uuid1>, <uuid2>],
   *  searchTerm: green st patricks day, green clothing
   * }
   */
  private parseQuery(smartSearchQuery: string): SmartSearchQuery {
    const similarToAssetIds: string[] = [];
    let searchText = smartSearchQuery;

    // Match all similarTo:<uuid> patterns
    const similarToRegex = /similarTo:([^\s]+)/gi;
    let match: RegExpExecArray | null;

    // Extract all asset IDs
    while ((match = similarToRegex.exec(smartSearchQuery)) !== null) {
      similarToAssetIds.push(match[1]);
      // Remove the matched pattern from the search text
      searchText = searchText.replace(match[0], '');
    }

    // Trim any extra whitespace and commas
    searchText = searchText
      .replace(/,\s*$/, '')
      .replace(/\s*,$/, '')
      .replaceAll(/\s{2,}/g, ' ') // multiple spaces -> 1 space
      .trim();

    return {
      similarToAssetIds,
      searchText,
    };
  }

  async searchSmart(auth: AuthDto, dto: SmartSearchDto): Promise<SearchResponseDto> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    if (!isSmartSearchEnabled(machineLearning)) {
      throw new BadRequestException('Smart search is not enabled');
    }

    const userIds = await this.getUserIdsToSearch(auth);
    const embedding = await this.machineLearningRepository.encodeText(machineLearning.urls, dto.query, {
      modelName: machineLearning.clip.modelName,
      language: dto.language,
    });
    const page = dto.page ?? 1;
    const size = dto.size || 100;
    const { hasNextPage, items } = await this.searchRepository.searchSmart(
      { page, size },
      { ...dto, userIds, embedding },
    );

    let items: AssetEntity[] = [];
    let hasNextPage = false;

    const searchBreakdown = this.parseQuery(dto.query);
    const searchEmbeddings: string[] = [];

    if (!searchBreakdown.searchText && !searchBreakdown.similarToAssetIds) {
      throw new BadRequestException('Search term is not understood');
    }

    if (searchBreakdown.similarToAssetIds) {
      (await this.searchRepository.getAssetEmbeddings(searchBreakdown.similarToAssetIds)).forEach((emb) =>
        searchEmbeddings.push(emb),
      );
    }

    if (searchBreakdown.searchText) {
      const textEmbedding = await this.machineLearningRepository.encodeText(
        machineLearning.urls,
        searchBreakdown.searchText,
        {
          modelName: machineLearning.clip.modelName,
          language: dto.language,
        },
      );
      searchEmbeddings.push(textEmbedding);
    }

    this.logger.log(
      `searchSmart; searchBreakdown=${JSON.stringify(searchBreakdown, null, 2)}; embeddings.length=${searchEmbeddings.length}`,
    );

    if (searchEmbeddings.length > 1) {
      const embedding = await this.machineLearningRepository.averageEmbeddings(machineLearning.urls, searchEmbeddings);
      const res = await this.searchRepository.searchSmart({ page, size }, { ...dto, userIds, embedding });
      items = res.items;
      hasNextPage = res.hasNextPage;
    } else {
      const res = await this.searchRepository.searchSmart(
        { page, size },
        { ...dto, userIds, embedding: searchEmbeddings[0] },
      );
      items = res.items;
      hasNextPage = res.hasNextPage;
    }

    return this.mapResponse(items, hasNextPage ? (page + 1).toString() : null, { auth });
  }

  async getAssetsByCity(auth: AuthDto): Promise<AssetResponseDto[]> {
    const userIds = await this.getUserIdsToSearch(auth);
    const assets = await this.searchRepository.getAssetsByCity(userIds);
    return assets.map((asset) => mapAsset(asset));
  }

  async getSearchSuggestions(auth: AuthDto, dto: SearchSuggestionRequestDto) {
    const userIds = await this.getUserIdsToSearch(auth);
    const suggestions = await this.getSuggestions(userIds, dto);
    if (dto.includeNull) {
      suggestions.push(null);
    }
    return suggestions;
  }

  private getSuggestions(userIds: string[], dto: SearchSuggestionRequestDto): Promise<Array<string | null>> {
    switch (dto.type) {
      case SearchSuggestionType.COUNTRY: {
        return this.searchRepository.getCountries(userIds);
      }
      case SearchSuggestionType.STATE: {
        return this.searchRepository.getStates(userIds, dto);
      }
      case SearchSuggestionType.CITY: {
        return this.searchRepository.getCities(userIds, dto);
      }
      case SearchSuggestionType.CAMERA_MAKE: {
        return this.searchRepository.getCameraMakes(userIds, dto);
      }
      case SearchSuggestionType.CAMERA_MODEL: {
        return this.searchRepository.getCameraModels(userIds, dto);
      }
      default: {
        return Promise.resolve([]);
      }
    }
  }

  private async getUserIdsToSearch(auth: AuthDto): Promise<string[]> {
    const partnerIds = await getMyPartnerIds({
      userId: auth.user.id,
      repository: this.partnerRepository,
      timelineEnabled: true,
    });
    return [auth.user.id, ...partnerIds];
  }

  private mapResponse(assets: AssetEntity[], nextPage: string | null, options: AssetMapOptions): SearchResponseDto {
    return {
      albums: { total: 0, count: 0, items: [], facets: [] },
      assets: {
        total: assets.length,
        count: assets.length,
        items: assets.map((asset) => mapAsset(asset, options)),
        facets: [],
        nextPage,
      },
    };
  }
}
